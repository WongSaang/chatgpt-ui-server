import os
import json
import openai
import datetime
import tiktoken
from .models import Conversation, Message, Setting
from django.http import StreamingHttpResponse
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from .serializers import ConversationSerializer, MessageSerializer


class ConversationViewSet(viewsets.ModelViewSet):
    serializer_class = ConversationSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).order_by('-created_at')


class MessageViewSet(viewsets.ModelViewSet):
    serializer_class = MessageSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Message.objects.filter(conversation_id=self.request.query_params.get('conversationId')).order_by('created_at')


@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def get_current_user(request):
    user = request.user
    return Response({
        'username': user.username,
    })


def sse_pack(event, data):
    # Format data as an SSE message
    packet = "event: %s\n" % event
    packet += "data: %s\n" % json.dumps(data)
    packet += "\n"
    return packet


@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def gen_title(request):
    conversation_id = request.data.get('conversationId')
    conversation = Conversation.objects.get(id=conversation_id)
    message = Message.objects.filter(conversation_id=conversation_id).order_by('created_at').first()
    prompt = f"Summarize this text with a concise title of 10 words or less(language by text):\n{message.message}"

    openai.api_key = get_openai_api_key()
    openai_response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0
    )
    completion_text = openai_response['choices'][0]['text']
    title = completion_text.strip().replace("\n", "")
    # update the conversation title
    conversation.topic = title
    conversation.save()
    return Response({
        'title': title
    })


@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def conversation(request):
    api_key = get_openai_api_key()
    if api_key is None:
        return Response({'error': 'The administrator has not set the API key'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    model = get_current_model()
    message = request.data.get('message')
    conversation_id = request.data.get('conversationId')
    parent_message_id = request.data.get('parentMessageId')

    if conversation_id:
        # get the conversation
        conversation_obj = Conversation.objects.get(id=conversation_id)
    else:
        # create a new conversation
        conversation_obj = Conversation(user=request.user)
        conversation_obj.save()
    # insert a new message
    message_obj = Message(
        conversation_id=conversation_obj.id,
        parent_message_id=parent_message_id,
        message=message
    )
    message_obj.save()

    prompt = build_prompt(conversation_obj)
    # print(prompt)

    num_tokens = get_token_count(prompt)
    max_tokens = min(model['max_tokens'] - num_tokens, model['max_response_tokens'])

    def stream_content():
        openai.api_key = api_key

        openai_response = openai.Completion.create(
            model=model['name'],
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.9,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=[" Human:", " AI:"],
            stream=True,
        )
        collected_events = []
        completion_text = ''
        # iterate through the stream of events
        for event in openai_response:
            collected_events.append(event)  # save the event response
            if event['choices'][0]['finish_reason'] == 'stop':
                ai_message_obj = Message(
                    conversation_id=conversation_obj.id,
                    parent_message_id=message_obj.id,
                    message=completion_text,
                    is_bot=True
                )
                ai_message_obj.save()
                yield sse_pack('done', {'messageId': ai_message_obj.id, 'conversationId': conversation_obj.id})
                break
            event_text = event['choices'][0]['text']  # extract the text
            completion_text += event_text  # append the text
            # print(event)
            yield sse_pack('message', {'content': event_text})

    return StreamingHttpResponse(stream_content(), content_type='text/event-stream')


def build_prompt(conversation_obj):
    model = get_current_model()

    ordered_messages = Message.objects.filter(conversation=conversation_obj).order_by('created_at')
    ordered_messages_list = list(ordered_messages)

    ai_label = 'AI'
    user_label = 'Human'
    separator_token = model['separator_token']
    current_date_string = datetime.datetime.today().strftime('%B %d, %Y')
    prompt_prefix = f"\n{separator_token}Instructions:\nYou are ChatGPT, a large language model trained by OpenAI.\nOutput the text in Markdown format{separator_token}\n\n"
    prompt_suffix = f"{ai_label}:\n"

    current_token_count = get_token_count(f"{prompt_prefix}{prompt_suffix}")
    prompt_body = ''
    max_token_count = model['max_prompt_tokens']

    while current_token_count < max_token_count and len(ordered_messages_list) > 0:
        message = ordered_messages_list.pop()
        role_label = ai_label if message.is_bot else user_label
        message_string = f"{role_label}:\n{message.message}{model['end_token']}\n"
        if prompt_body:
            new_prompt_body = f"{message_string}{prompt_body}"
        else:
            new_prompt_body = f"{prompt_prefix}{message_string}{prompt_body}"

        new_token_count = get_token_count(f"{prompt_prefix}{new_prompt_body}{prompt_suffix}")
        if new_token_count > max_token_count:
            if prompt_body:
                break
            raise ValueError(f"Prompt is too long. Max token count is {max_token_count}, but prompt is {new_token_count} tokens long.")
        prompt_body = new_prompt_body
        current_token_count = new_token_count

    prompt = f"{prompt_body}{prompt_suffix}"

    return prompt

def get_current_model():
    model = {
        'name': 'text-davinci-003',
        'max_tokens': 4097,
        'max_prompt_tokens': 3097,
        'max_response_tokens': 1000,
        'separator_token': '<|im_sep|>',
        'end_token': '<|im_end|>'
    }
    return model

def get_openai_api_key():
    row = Setting.objects.filter(name='openai_api_key').first()
    if row:
        return row.value
    return None

def get_token_count(token):
    model = get_current_model()
    enc = tiktoken.encoding_for_model(model['name'])
    return len(enc.encode(token))